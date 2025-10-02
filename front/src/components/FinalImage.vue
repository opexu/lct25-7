<template>
<div class="relative w-full h-full">
    <img 
    class="absolute left-0 top-0 w-full h-full object-cover" 
    :class="{ 'blurry': src }"
    :src="src"
    @error="onError"
    />
    <div class="absolute w-full h-full left-0 top-0 p-8 flex flex-row items-end justify-center"
    v-if="showBtns"
    >
        <div class="w-full h-fit gap-4 flex flex-col items-center justify-start">
            <button class="w-fit h-fit px-8 py-4 border rounded-2xl text-secondary"
            @click="$emit('back')"
            >Попробовать снова</button>
            <router-link class="w-fit h-fit px-8 py-4 border rounded-2xl text-primary"
            to="/" @click="vibro.vibrate([50,50,50])"
            >На главную</router-link>
        </div>
    </div>
</div>
</template>

<script setup lang="ts">
import { useVibration } from '@/composables/useVibration';
import { onMounted, onUnmounted, ref } from 'vue';
import { useRouter } from 'vue-router';
const vibro = useVibration()

const router = useRouter()
defineEmits(['error','back'])
const props = defineProps<{
    src: string
}>()

const showBtns = ref(false)
let timeout: number
onMounted(() => {
    if( timeout ) clearTimeout( timeout )
    timeout = setTimeout(() => {
        showBtns.value = true
        // onError()
    }, 2000)
})
onUnmounted(() => {
    if( timeout ) clearTimeout( timeout )
})

function onError(){
    router.push({ name: 'error', params: {
        msg: 'Что-то пошло не так',
        showMainBtn: '1',
    }})
}
</script>

<style scoped>
@keyframes blurry {
  0% {
    filter: blur(20px);
    animation-timing-function: cubic-bezier(0.8, 0, 1, 1);
  }
  100% {
    filter: blur(0);
    animation-timing-function: cubic-bezier(0, 0, 0.2, 1);
  }
}
.blurry {
    animation: blurry ease-in-out 3s
}
</style>