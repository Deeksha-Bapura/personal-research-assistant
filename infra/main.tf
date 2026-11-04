# Runs the dashboard + multi-agent API as Docker containers on your own
# machine, managed by Terraform. No cloud account and no cost.
#
#   cd infra
#   terraform init
#   terraform apply
#   -> open the dashboard_url output
#   terraform destroy   # stop and remove everything

provider "docker" {
  host = var.docker_host != "" ? var.docker_host : null
}

locals {
  backend_dir  = abspath("${path.module}/../backend")
  frontend_dir = abspath("${path.module}/../frontend")

  # KEY=VALUE lines from backend/.env (comments and blank lines dropped;
  # trimspace also strips Windows \r line endings).
  # Wrapped in sensitive() so plans and logs never print your API keys.
  api_env = sensitive([
    for line in split("\n", file("${path.module}/${var.env_file}")) :
    trimspace(line)
    if trimspace(line) != "" && !startswith(trimspace(line), "#")
  ])

  # Rebuild an image only when the files that go into it change.
  backend_hash = sha1(join("", [
    for f in sort(setunion(
      fileset(local.backend_dir, "*.py"),
      fileset(local.backend_dir, "agents/*.py"),
      fileset(local.backend_dir, "{requirements.txt,Dockerfile,.dockerignore}"),
    )) : filesha1("${local.backend_dir}/${f}")
  ]))

  frontend_hash = sha1(join("", [
    for f in sort(setunion(
      fileset(local.frontend_dir, "src/**"),
      fileset(local.frontend_dir, "public/**"),
      fileset(local.frontend_dir, "{index.html,package.json,package-lock.json,vite.config.ts,tsconfig*.json,Dockerfile,nginx.conf.template}"),
    )) : filesha1("${local.frontend_dir}/${f}")
  ]))
}

resource "docker_network" "app" {
  name = "${var.name_prefix}-net"
}

# ---------------------------------------------------------------- images

resource "docker_image" "backend" {
  name = "${var.name_prefix}-backend:tf"

  build {
    context = local.backend_dir
  }

  triggers = {
    source_hash = local.backend_hash
  }
}

resource "docker_image" "frontend" {
  name = "${var.name_prefix}-frontend:tf"

  build {
    context = local.frontend_dir
  }

  triggers = {
    source_hash = local.frontend_hash
  }
}

# ------------------------------------------------------------ containers

resource "docker_container" "backend" {
  name    = "${var.name_prefix}-backend"
  image   = docker_image.backend.image_id
  restart = "unless-stopped"
  env     = local.api_env

  networks_advanced {
    name    = docker_network.app.name
    aliases = ["backend"]
  }

  ports {
    internal = 8000
    external = var.backend_port
  }

  # Wait for the image's HEALTHCHECK (/health) before starting the frontend
  wait         = true
  wait_timeout = 120
}

resource "docker_container" "frontend" {
  name    = "${var.name_prefix}-frontend"
  image   = docker_image.frontend.image_id
  restart = "unless-stopped"
  env     = ["API_UPSTREAM=http://backend:8000"]

  networks_advanced {
    name = docker_network.app.name
  }

  ports {
    internal = 80
    external = var.frontend_port
  }

  depends_on = [docker_container.backend]
}
