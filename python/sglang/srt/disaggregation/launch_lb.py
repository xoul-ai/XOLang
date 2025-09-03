import argparse
import dataclasses

from sglang.srt.disaggregation.mini_lb import PrefillConfig, run
from sglang.srt.disaggregation.service_discovery import ServiceDiscoveryConfig


@dataclasses.dataclass
class LBArgs:
    host: str = "0.0.0.0"
    port: int = 8000
    policy: str = "random"
    prefill_infos: list = dataclasses.field(default_factory=list)
    decode_infos: list = dataclasses.field(default_factory=list)
    log_interval: int = 5
    timeout: int = 600
    # Service discovery configuration
    service_discovery: bool = False
    prefill_selector: dict = dataclasses.field(default_factory=dict)
    decode_selector: dict = dataclasses.field(default_factory=dict)
    service_discovery_port: int = 8000
    service_discovery_namespace: str = None
    bootstrap_port_annotation: str = "sglang.ai/bootstrap-port"
    service_discovery_check_interval: int = 60

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--host",
            type=str,
            default=LBArgs.host,
            help=f"Host to bind the server (default: {LBArgs.host})",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=LBArgs.port,
            help=f"Port to bind the server (default: {LBArgs.port})",
        )
        parser.add_argument(
            "--policy",
            type=str,
            default=LBArgs.policy,
            choices=["random", "po2"],
            help=f"Policy to use for load balancing (default: {LBArgs.policy})",
        )
        parser.add_argument(
            "--prefill",
            type=str,
            default=[],
            nargs="+",
            help="URLs for prefill servers",
        )
        parser.add_argument(
            "--decode",
            type=str,
            default=[],
            nargs="+",
            help="URLs for decode servers",
        )
        parser.add_argument(
            "--prefill-bootstrap-ports",
            type=int,
            nargs="+",
            help="Bootstrap ports for prefill servers",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=LBArgs.log_interval,
            help=f"Log interval in seconds (default: {LBArgs.log_interval})",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=LBArgs.timeout,
            help=f"Timeout in seconds (default: {LBArgs.timeout})",
        )
        # Service discovery arguments
        parser.add_argument(
            "--service-discovery",
            action="store_true",
            help="Enable Kubernetes service discovery",
        )
        parser.add_argument(
            "--prefill-selector",
            type=str,
            nargs="+",
            help="Label selector for prefill server pods (format: key1=value1 key2=value2)",
        )
        parser.add_argument(
            "--decode-selector",
            type=str,
            nargs="+",
            help="Label selector for decode server pods (format: key1=value1 key2=value2)",
        )
        parser.add_argument(
            "--service-discovery-port",
            type=int,
            default=LBArgs.service_discovery_port,
            help=f"Port to use for discovered worker pods (default: {LBArgs.service_discovery_port})",
        )
        parser.add_argument(
            "--service-discovery-namespace",
            type=str,
            help="Kubernetes namespace to watch for pods. If not provided, watches all namespaces",
        )
        parser.add_argument(
            "--bootstrap-port-annotation",
            type=str,
            default=LBArgs.bootstrap_port_annotation,
            help=f"Annotation key for bootstrap port (default: {LBArgs.bootstrap_port_annotation})",
        )
        parser.add_argument(
            "--service-discovery-check-interval",
            type=int,
            default=LBArgs.service_discovery_check_interval,
            help=f"Interval in seconds between service discovery checks (default: {LBArgs.service_discovery_check_interval})",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "LBArgs":
        bootstrap_ports = args.prefill_bootstrap_ports
        if bootstrap_ports is None:
            bootstrap_ports = [None] * len(args.prefill)
        elif len(bootstrap_ports) == 1:
            bootstrap_ports = bootstrap_ports * len(args.prefill)
        else:
            if len(bootstrap_ports) != len(args.prefill):
                raise ValueError(
                    "Number of prefill URLs must match number of bootstrap ports"
                )

        prefill_infos = [
            (url, port) for url, port in zip(args.prefill, bootstrap_ports)
        ]

        # Parse service discovery selectors
        prefill_selector = cls._parse_selector(getattr(args, 'prefill_selector', None))
        decode_selector = cls._parse_selector(getattr(args, 'decode_selector', None))

        return cls(
            host=args.host,
            port=args.port,
            policy=args.policy,
            prefill_infos=prefill_infos,
            decode_infos=args.decode,
            log_interval=args.log_interval,
            timeout=args.timeout,
            service_discovery=getattr(args, 'service_discovery', False),
            prefill_selector=prefill_selector,
            decode_selector=decode_selector,
            service_discovery_port=getattr(args, 'service_discovery_port', LBArgs.service_discovery_port),
            service_discovery_namespace=getattr(args, 'service_discovery_namespace', None),
            bootstrap_port_annotation=getattr(args, 'bootstrap_port_annotation', LBArgs.bootstrap_port_annotation),
            service_discovery_check_interval=getattr(args, 'service_discovery_check_interval', LBArgs.service_discovery_check_interval),
        )

    @staticmethod
    def _parse_selector(selector_list):
        """Parse label selector from command line arguments."""
        if not selector_list:
            return {}

        selector = {}
        for item in selector_list:
            if "=" in item:
                key, value = item.split("=", 1)
                selector[key] = value
        return selector


def main():
    parser = argparse.ArgumentParser(
        description="PD Disaggregation Load Balancer Server with Kubernetes Service Discovery"
    )
    LBArgs.add_cli_args(parser)
    args = parser.parse_args()
    lb_args = LBArgs.from_cli_args(args)

    # Validate service discovery configuration
    if lb_args.service_discovery:
        if not lb_args.prefill_selector and not lb_args.decode_selector:
            raise ValueError("Service discovery enabled but no selectors configured. "
                           "Please specify --prefill-selector and/or --decode-selector")
        
        print(f"Service discovery enabled:")
        if lb_args.prefill_selector:
            print(f"  Prefill selector: {lb_args.prefill_selector}")
        if lb_args.decode_selector:
            print(f"  Decode selector: {lb_args.decode_selector}")
        print(f"  Service discovery port: {lb_args.service_discovery_port}")
        if lb_args.service_discovery_namespace:
            print(f"  Namespace: {lb_args.service_discovery_namespace}")
        else:
            print("  Namespace: all namespaces")

    # Create service discovery configuration
    service_discovery_config = None
    if lb_args.service_discovery:
        service_discovery_config = ServiceDiscoveryConfig(
            enabled=True,
            prefill_selector=lb_args.prefill_selector,
            decode_selector=lb_args.decode_selector,
            service_discovery_port=lb_args.service_discovery_port,
            namespace=lb_args.service_discovery_namespace,
            bootstrap_port_annotation=lb_args.bootstrap_port_annotation,
            check_interval=lb_args.service_discovery_check_interval,
        )

    prefill_configs = [PrefillConfig(url, port) for url, port in lb_args.prefill_infos]
    run(
        prefill_configs,
        lb_args.decode_infos,
        lb_args.host,
        lb_args.port,
        lb_args.timeout,
        service_discovery_config,
    )


if __name__ == "__main__":
    main()
