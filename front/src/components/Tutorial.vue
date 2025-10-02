<template>
<div class="relative flex flex-row items-center justify-center">
    <transition name="fade" mode="out-in">
        <div class="relative w-fit h-fit"
        :key="tutorialStep"
        >
            <img class="w-full h-full object-cover" 
            :src="baseImage"
            :srcset="srcSet" 
            @load="onLoad" 
            @error="onError" 
            />
            <button class="absolute bottom-0 w-full h-16"
            @click="nextStep"
            ></button>
        </div>
    </transition>
</div>
</template>

<script setup lang="ts">
import { useVibration } from '@/composables/useVibration';
import * as Utils from '@/scripts/utils';
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useRouter } from 'vue-router';

const vibro = useVibration()
const tutorialStep = ref(1)

const baseImage = computed(() => getImagePath( tutorialStep.value, 1 ))
const srcSet = computed(() => genSrcset( tutorialStep.value, 3 ))
// const srcSet = ref( genSrcset( tutorialStep.value, 3 ) )
function getImagePath( index: number, set: number ): string {
	return `/tutorial/tut-${index}-${set}.png ${set}x`
}

function genSrcset( image: number, count: number ): string {
	const arr = Utils.genRange( 1, count )
	return arr.map( num => getImagePath( image, num ) ).join( ',' )
}

function onLoad() {
	console.log( 'on load tutorial' )
}

function onError( e: Event ) {
	console.log( 'on error tutorial: ', e )
}

const router = useRouter()
function nextStep(){
    tutorialStep.value += 1
    vibro.vibrate([50])
    if( tutorialStep.value >= 5 ){
        router.push('/camera')
        tutorialStep.value = 1
        vibro.vibrate([50,50,50])
    }
}
</script>