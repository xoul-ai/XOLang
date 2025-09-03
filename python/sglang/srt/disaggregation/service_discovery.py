"""
Kubernetes service discovery for mini-lb (PD disaggregation load balancer).
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

logger = logging.getLogger("mini_lb_service_discovery")


@dataclass
class ServiceDiscoveryConfig:
    """Configuration for Kubernetes service discovery."""
    enabled: bool = False
    prefill_selector: Dict[str, str] = None
    decode_selector: Dict[str, str] = None
    service_discovery_port: int = 8000
    namespace: Optional[str] = None
    bootstrap_port_annotation: str = "sglang.ai/bootstrap-port"
    check_interval: int = 60

    def __post_init__(self):
        if self.prefill_selector is None:
            self.prefill_selector = {}
        if self.decode_selector is None:
            self.decode_selector = {}


@dataclass
class PodInfo:
    """Information about a discovered pod."""
    name: str
    ip: str
    status: str
    is_ready: bool
    pod_type: str  # "prefill" or "decode"
    bootstrap_port: Optional[int] = None

    def is_healthy(self) -> bool:
        """Check if pod is healthy and ready to accept traffic."""
        return self.is_ready and self.status == "Running"

    def get_worker_url(self, port: int) -> str:
        """Generate worker URL for this pod."""
        return f"http://{self.ip}:{port}"


class ServiceDiscovery:
    """Kubernetes service discovery for mini-lb."""

    def __init__(self, config: ServiceDiscoveryConfig, load_balancer):
        self.config = config
        self.load_balancer = load_balancer
        self.tracked_pods: Set[PodInfo] = set()
        self.k8s_client = None
        self.watcher_task = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the service discovery process."""
        if not self.config.enabled:
            logger.info("Service discovery is disabled")
            return

        try:
            # Load Kubernetes configuration
            try:
                config.load_incluster_config()  # Try in-cluster config first
                logger.info("Using in-cluster Kubernetes configuration")
            except config.ConfigException:
                try:
                    config.load_kube_config()  # Fall back to local config
                    logger.info("Using local Kubernetes configuration")
                except config.ConfigException as e:
                    logger.error(f"Failed to load Kubernetes configuration: {e}")
                    return

            self.k8s_client = client.CoreV1Api()

            # Validate selectors
            if not self.config.prefill_selector and not self.config.decode_selector:
                logger.warning("Service discovery enabled but no selectors configured")
                return

            # Log configuration
            if self.config.prefill_selector:
                prefill_selector_str = ", ".join(f"{k}={v}" for k, v in self.config.prefill_selector.items())
                logger.info(f"Service discovery: watching prefill pods with selector: {prefill_selector_str}")
            
            if self.config.decode_selector:
                decode_selector_str = ", ".join(f"{k}={v}" for k, v in self.config.decode_selector.items())
                logger.info(f"Service discovery: watching decode pods with selector: {decode_selector_str}")

            # Start the watcher task
            self.watcher_task = asyncio.create_task(self._watch_pods())
            logger.info("Service discovery started successfully")

        except Exception as e:
            logger.error(f"Failed to start service discovery: {e}")

    async def stop(self):
        """Stop the service discovery process."""
        if self.watcher_task:
            self.watcher_task.cancel()
            try:
                await self.watcher_task
            except asyncio.CancelledError:
                pass
        logger.info("Service discovery stopped")

    async def _watch_pods(self):
        """Watch for pod changes and update the load balancer."""
        retry_delay = 1
        max_retry_delay = 300  # 5 minutes

        while True:
            try:
                # Create label selector
                label_selector = self._create_label_selector()
                
                # Watch pods
                w = watch.Watch()
                for event in w.stream(
                    self.k8s_client.list_pod_for_all_namespaces,
                    label_selector=label_selector,
                    timeout_seconds=self.config.check_interval
                ):
                    pod = event['object']
                    event_type = event['type']
                    
                    await self._handle_pod_event(pod, event_type)

                # Reset retry delay on successful watch
                retry_delay = 1

            except ApiException as e:
                logger.error(f"Kubernetes API error: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
            except Exception as e:
                logger.error(f"Error in pod watcher: {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)

    def _create_label_selector(self) -> str:
        """Create Kubernetes label selector from configuration."""
        selectors = []
        
        if self.config.prefill_selector:
            prefill_selector = ",".join(f"{k}={v}" for k, v in self.config.prefill_selector.items())
            selectors.append(f"({prefill_selector})")
        
        if self.config.decode_selector:
            decode_selector = ",".join(f"{k}={v}" for k, v in self.config.decode_selector.items())
            selectors.append(f"({decode_selector})")
        
        return " || ".join(selectors) if len(selectors) > 1 else selectors[0] if selectors else ""

    async def _handle_pod_event(self, pod, event_type: str):
        """Handle a pod event (ADDED, MODIFIED, DELETED)."""
        try:
            pod_info = self._extract_pod_info(pod)
            if not pod_info:
                return

            async with self._lock:
                if event_type in ['ADDED', 'MODIFIED']:
                    await self._handle_pod_added_or_modified(pod_info)
                elif event_type == 'DELETED':
                    await self._handle_pod_deleted(pod_info)

        except Exception as e:
            logger.error(f"Error handling pod event: {e}")

    def _extract_pod_info(self, pod) -> Optional[PodInfo]:
        """Extract PodInfo from Kubernetes pod object."""
        try:
            # Get basic pod information
            name = pod.metadata.name
            ip = pod.status.pod_ip
            status = pod.status.phase

            if not ip:
                return None

            # Check if pod is ready
            is_ready = False
            if pod.status.conditions:
                for condition in pod.status.conditions:
                    if condition.type == "Ready" and condition.status == "True":
                        is_ready = True
                        break

            # Determine pod type based on labels
            pod_type = None
            bootstrap_port = None

            if self._pod_matches_selector(pod, self.config.prefill_selector):
                pod_type = "prefill"
                # Extract bootstrap port from annotations
                if pod.metadata.annotations:
                    bootstrap_port_str = pod.metadata.annotations.get(self.config.bootstrap_port_annotation)
                    if bootstrap_port_str:
                        try:
                            bootstrap_port = int(bootstrap_port_str)
                        except ValueError:
                            logger.warning(f"Invalid bootstrap port annotation for pod {name}: {bootstrap_port_str}")
            elif self._pod_matches_selector(pod, self.config.decode_selector):
                pod_type = "decode"

            if not pod_type:
                return None

            return PodInfo(
                name=name,
                ip=ip,
                status=status,
                is_ready=is_ready,
                pod_type=pod_type,
                bootstrap_port=bootstrap_port
            )

        except Exception as e:
            logger.error(f"Error extracting pod info: {e}")
            return None

    def _pod_matches_selector(self, pod, selector: Dict[str, str]) -> bool:
        """Check if pod matches the given selector."""
        if not selector or not pod.metadata.labels:
            return False

        for key, value in selector.items():
            if pod.metadata.labels.get(key) != value:
                return False
        return True

    async def _handle_pod_added_or_modified(self, pod_info: PodInfo):
        """Handle pod addition or modification."""
        # Check if pod is already tracked
        existing_pod = None
        for tracked_pod in self.tracked_pods:
            if tracked_pod.name == pod_info.name:
                existing_pod = tracked_pod
                break

        # Only add healthy pods
        if pod_info.is_healthy():
            if not existing_pod:
                # New healthy pod - add to load balancer
                await self._add_pod_to_load_balancer(pod_info)
                self.tracked_pods.add(pod_info)
                logger.info(f"Added {pod_info.pod_type} pod: {pod_info.name} ({pod_info.ip})")
            else:
                # Pod was already tracked, check if it needs updating
                if existing_pod.ip != pod_info.ip or existing_pod.bootstrap_port != pod_info.bootstrap_port:
                    # Pod IP or bootstrap port changed - remove old and add new
                    await self._remove_pod_from_load_balancer(existing_pod)
                    await self._add_pod_to_load_balancer(pod_info)
                    self.tracked_pods.remove(existing_pod)
                    self.tracked_pods.add(pod_info)
                    logger.info(f"Updated {pod_info.pod_type} pod: {pod_info.name} ({pod_info.ip})")
        else:
            # Pod is not healthy - remove if it was tracked
            if existing_pod:
                await self._remove_pod_from_load_balancer(existing_pod)
                self.tracked_pods.remove(existing_pod)
                logger.info(f"Removed unhealthy {existing_pod.pod_type} pod: {existing_pod.name}")

    async def _handle_pod_deleted(self, pod_info: PodInfo):
        """Handle pod deletion."""
        # Find and remove the pod
        for tracked_pod in list(self.tracked_pods):
            if tracked_pod.name == pod_info.name:
                await self._remove_pod_from_load_balancer(tracked_pod)
                self.tracked_pods.remove(tracked_pod)
                logger.info(f"Removed deleted {tracked_pod.pod_type} pod: {tracked_pod.name}")
                break

    async def _add_pod_to_load_balancer(self, pod_info: PodInfo):
        """Add a pod to the load balancer."""
        try:
            worker_url = pod_info.get_worker_url(self.config.service_discovery_port)
            
            if pod_info.pod_type == "prefill":
                from sglang.srt.disaggregation.mini_lb import PrefillConfig
                prefill_config = PrefillConfig(worker_url, pod_info.bootstrap_port)
                self.load_balancer.add_prefill_server(prefill_config)
            elif pod_info.pod_type == "decode":
                self.load_balancer.add_decode_server(worker_url)
            
            logger.debug(f"Added {pod_info.pod_type} server: {worker_url}")
            
        except Exception as e:
            logger.error(f"Failed to add {pod_info.pod_type} pod {pod_info.name} to load balancer: {e}")

    async def _remove_pod_from_load_balancer(self, pod_info: PodInfo):
        """Remove a pod from the load balancer."""
        try:
            worker_url = pod_info.get_worker_url(self.config.service_discovery_port)
            
            if pod_info.pod_type == "prefill":
                # Find and remove the prefill server
                for i, config in enumerate(self.load_balancer.prefill_configs):
                    if config.url == worker_url:
                        self.load_balancer.prefill_configs.pop(i)
                        self.load_balancer.prefill_servers.remove(worker_url)
                        break
            elif pod_info.pod_type == "decode":
                if worker_url in self.load_balancer.decode_servers:
                    self.load_balancer.decode_servers.remove(worker_url)
            
            logger.debug(f"Removed {pod_info.pod_type} server: {worker_url}")
            
        except Exception as e:
            logger.error(f"Failed to remove {pod_info.pod_type} pod {pod_info.name} from load balancer: {e}")

    def get_status(self) -> Dict:
        """Get current service discovery status."""
        return {
            "enabled": self.config.enabled,
            "tracked_pods": len(self.tracked_pods),
            "prefill_servers": len(self.load_balancer.prefill_servers),
            "decode_servers": len(self.load_balancer.decode_servers),
            "pod_details": [
                {
                    "name": pod.name,
                    "ip": pod.ip,
                    "type": pod.pod_type,
                    "status": pod.status,
                    "is_ready": pod.is_ready,
                    "bootstrap_port": pod.bootstrap_port
                }
                for pod in self.tracked_pods
            ]
        }
