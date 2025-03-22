# quant-trading-system
A simple implementation of the quant trading system. You can build it by issue command: 'docker compose --build' under the directory where docker-compose.yml file lies, and start it by issue command: 'docker compose up -d' for a detached mode. The default listening port is set for 8080, you can set this as you wish.

# Docker hub mirror -- You can execute this command on the linux shell if you can not access docker hub.
```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
    "registry-mirrors": [
        "https://docker.1ms.run",
        "https://docker.xuanyuan.me"
    ]
}
EOF

```

# If the package install process is slow, you can change to the mirror repository.
