FROM nginx:alpine
LABEL maintainer="vanshtiwari1404"

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy static frontend files (index.html, index.css, images)
COPY . /usr/share/nginx/html

EXPOSE 80