variable "docker_host" {
  description = "Docker daemon to talk to. Leave empty for the default socket (Linux/macOS). On Windows with Docker Desktop use \"npipe:////.//pipe/docker_engine\"."
  type        = string
  default     = ""
}

variable "env_file" {
  description = "Path (relative to infra/) of the .env file whose KEY=VALUE lines are passed to the API container."
  type        = string
  default     = "../backend/.env"
}

variable "frontend_port" {
  description = "Host port for the dashboard."
  type        = number
  default     = 8080
}

variable "backend_port" {
  description = "Host port for the API (handy for /health; the dashboard reaches it over the Docker network)."
  type        = number
  default     = 8000
}

variable "name_prefix" {
  description = "Prefix for container, network and image names."
  type        = string
  default     = "research-assistant"
}
