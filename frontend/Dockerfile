# ── BurnOut Lens — Frontend Dockerfile ────────────────────────────────────
#
# Multi-stage:
#   Stage 1 (builder): Node 22 alpine — npm ci + vite build
#   Stage 2 (runner):  nginx alpine    — serves compiled static files
#
# VITE_API_BASE_URL is a BUILD ARG — Vite bakes it into the compiled JS at
# build time.  It cannot be a runtime env var because the JS runs in the
# browser, not in the container.  The value must be the URL reachable from
# the end-user's browser (i.e. http://localhost:8000 for local dev), NOT
# the Docker-internal service name (which only works inside the Docker network).

# ── Stage 1: build ────────────────────────────────────────────────────────
FROM node:22-alpine AS builder

WORKDIR /app

# Copy manifests first for layer caching — npm ci only re-runs when these change
COPY package.json package-lock.json ./

RUN npm ci --silent

# Build arg for the backend URL (baked into the JS bundle at build time)
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# Copy source and build
COPY . .
RUN npm run build

# ── Stage 2: serve (non-root) ─────────────────────────────────────────────
# The official nginx:alpine image ships with a built-in 'nginx' user.
# To run as non-root we: (1) listen on unprivileged port 8080 instead of 80,
# (2) chown the dirs nginx needs to write (cache, logs, pid), (3) move the
# PID file to /tmp so nginx can create it without root.
FROM nginx:1.27-alpine AS runner

# Give the nginx user ownership of directories it needs at runtime
RUN rm -rf /usr/share/nginx/html/* && \
    chown -R nginx:nginx /var/cache/nginx /var/log/nginx && \
    touch /tmp/nginx.pid && chown nginx:nginx /tmp/nginx.pid

# Patch the main nginx.conf: PID file → /tmp (writable by nginx user)
RUN sed -i 's|/run/nginx.pid|/tmp/nginx.pid|' /etc/nginx/nginx.conf

# Copy the built app from builder stage
COPY --from=builder --chown=nginx:nginx /app/dist /usr/share/nginx/html

# nginx config: serve index.html as fallback for SPA routes
# (not strictly needed now since there are no client-side routes, but
#  correct practice and costs nothing — prevents 404 on direct URL access)
COPY --chown=nginx:nginx nginx.conf /etc/nginx/conf.d/default.conf

USER nginx

EXPOSE 8080

CMD ["nginx", "-g", "daemon off;"]
