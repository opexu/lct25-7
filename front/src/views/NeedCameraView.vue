<template>
    <div class="w-full h-full p-8 gap-4 flex flex-col items-center justify-center gradient">
        <div class="w-full h-full">
            <img class="h-full object-cover" src="/child.gif" alt="gif" draggable="false">
        </div>
        <p class="text-primary font-bold text-4xl text-center">Приложению требуется доступ к камере для того, чтобы сделать фото</p>
        <button class="w-fit h-fit px-8 py-4 border rounded-2xl text-secondary"
        @click="$emit('req-camera-permissions')"
        >Разрешить камеру</button>
        <router-link class="w-fit h-fit px-8 py-4 border rounded-2xl text-primary"
        to="/" @click="vibro.vibrate([50,50,50])"
        >На главную</router-link>
    </div>
</template>

<script setup lang="ts">
import { useVibration } from '@/composables/useVibration';

defineEmits(['req-camera-permissions'])
const vibro = useVibration()
</script>