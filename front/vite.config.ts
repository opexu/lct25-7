import { fileURLToPath, URL } from 'node:url'
import fs from 'fs';
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import tailwindcss from '@tailwindcss/vite'
import { resolve, join } from 'path';

// https://vite.dev/config/
export default defineConfig( ( { mode } ) => {
	const isDev = mode === 'development'
	const plugins = isDev ? [
		vue(),
		vueDevTools(),
		tailwindcss(),
	] : [ vue(), tailwindcss(), ]
	return {
		plugins,
		resolve: {
			alias: {
				'@': fileURLToPath( new URL( './src', import.meta.url ) )
			},
		},
		...( isDev && {
			server: {
				port: 5173,
				host: '0.0.0.0',
				// https: {
				// 	key: fs.readFileSync( resolve( __dirname, 'key.pem' ) ),
				// 	cert: fs.readFileSync( resolve( __dirname, 'cert.pem' ) ),
				// },
				allowedHosts: true,
			},
		} )
	}
} )
