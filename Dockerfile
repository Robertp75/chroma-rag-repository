FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates iptables \
  && rm -rf /var/lib/apt/lists/*

# Install tailscale
RUN curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg \
    | tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null && \
    curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list \
    | tee /etc/apt/sources.list.d/tailscale.list >/dev/null && \
    apt-get update && apt-get install -y tailscale && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
