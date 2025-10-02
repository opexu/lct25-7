FROM node:22-alpine as front
WORKDIR /front

COPY --chown=node:node \
    ./package.json \
    ./package-lock.json \
    ./tsconfig.json \
    ./tsconfig.app.json \
    ./tsconfig.node.json \
    ./vite.config.ts \
    ./index.html \
    ./

RUN npm ci

# COPY --chown=node:node \
#     ./src \
#     ./src

# COPY --chown=node:node \
#     ./public \
#     ./public

ARG VITE_NODE_ENV
ENV VITE_NODE_ENV=${VITE_NODE_ENV}

CMD ["sh","-c","npm run dev"]