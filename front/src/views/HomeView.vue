<template>
	<div class="relative w-full h-full flex flex-row items-center justify-center">
		<BackgroundImage>
			<div class="w-full h-full flex flex-col items-center">
				<img class="w-70" src="/logo.svg"></img>
				<Tutorial class="w-full h-fit"/>
			</div>
		</BackgroundImage>

	</div>
</template>

<script setup lang="ts">
import { usePreloader } from '@/stores/preloader';
import { storeToRefs } from 'pinia';
import BackgroundImage from '@/components/BackgroundImage.vue';
import Tutorial from '@/components/Tutorial.vue';
import { onBeforeMount } from 'vue';
import { useDevice } from '@/stores/device';
import { useRouter } from 'vue-router';

const preloaderStore = usePreloader()
const { active } = storeToRefs( preloaderStore )
const deviceStore = useDevice()
const router = useRouter()
onBeforeMount(() => {
	if( !deviceStore.isDesktop ) return
	router.push({ name: 'error', params: {
        msg: 'Ой, это мобильное приложение',
    }})
})
</script>