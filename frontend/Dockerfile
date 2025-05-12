#Build the React application
FROM node:20-alpine AS build-stage

LABEL maintainer="your-name-or-email@example.com"
LABEL description="Build stage for the RAG frontend React application"

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the application (output will be in /app/dist)
RUN npm run build

# Serve the application with Nginx
FROM nginx:stable-alpine AS serve-stage

# Copy built assets from the build-stage
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]