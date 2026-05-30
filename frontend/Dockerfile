# ── Stage 1: build the Vite/React app ────────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

# Install dependencies first (better layer caching)
COPY package.json package-lock.json ./
RUN npm ci

# Build (runs `tsc && vite build` → static assets in /app/dist)
COPY . .
RUN npm run build

# ── Stage 2: serve static assets via nginx + reverse-proxy the API ────────────
FROM nginx:1.27-alpine AS serve

# SPA + /it-kb and /health reverse proxy to the backend container
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
