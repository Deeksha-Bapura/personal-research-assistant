output "dashboard_url" {
  description = "Open this in your browser."
  value       = "http://localhost:${var.frontend_port}"
}

output "api_health_url" {
  description = "API health check."
  value       = "http://localhost:${var.backend_port}/health"
}

output "containers" {
  value = [docker_container.backend.name, docker_container.frontend.name]
}
