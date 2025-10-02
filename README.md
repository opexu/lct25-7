
# AR-фотозона

web приложение, которое добавляет эффекты к фотографиям




## Tech Stack

**Client:** Vite, Vue 3 Composition API, Pinia, TailwindCSS

**Backend:** python, fastapi

**Server:** traefik


## Run Locally

1. Clone the project
```bash
  git clone https://github.com/opexu/lct25-7.git
```

2. Go to the project directory
```bash
  cd lct25-7
```

3. Create .env
```bash
  VITE_NODE_ENV=dev
```
4. Start app 
( Requires: Docker Compose 2.20.3 and later )
```bash
  docker compose up -d --build
```
5. Wait docker build ( 20 minutes )

6. Go to "http://localhost" from mobile device from same network