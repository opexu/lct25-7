<template>
	<div class="relative w-full h-full">
		<div class="w-full h-full">
			<transition name="fade-slide-zoom-slow" mode="in-out">
				<img class="absolute left-0 top-0 w-full h-full object-cover blur-sm" :key="srcSet1" :src="baseImage1"
					:srcset="srcSet1" @load="onLoad" @error="onError" />
			</transition>
		</div>
		<div class="absolute w-full h-full left-0 top-0">
			<slot></slot>
		</div>
	</div>
</template>

<script setup lang="ts">
import * as Utils from '@/scripts/utils';
import { onMounted, onUnmounted, ref } from 'vue';

const imgIndexes = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ]
const time = 5000

const imgIndex = Utils.getRandomFromRange( imgIndexes )
const baseImage1 = ref( getImagePath( imgIndex, 1 ) )
const srcSet1 = ref( genSrcset( imgIndex, 3 ) )
console.log( 'srcSet1: ', srcSet1.value )

let interval: number
onMounted( () => {
	interval = setInterval( () => {
		if ( inProgress ) return
		const imgIndex = Utils.getRandomFromRange( imgIndexes )
		baseImage1.value = getImagePath( imgIndex, 1 )
		srcSet1.value = genSrcset( imgIndex, 3 )
		inProgress = true
	}, time )
} )

onUnmounted( () => {
	if ( interval ) {
		clearInterval( interval )
	}
} )
function getImagePath( index: number, set: number ): string {
	return `/bg/${index}/bg-${index}-${set}x.jpg ${set}x`
}

function genSrcset( image: number, count: number ): string {
	const arr = Utils.genRange( 1, count )
	return arr.map( num => getImagePath( image, num ) ).join( ',' )
}

let inProgress = false
function onLoad() {
	console.log( 'on load' )
	inProgress = false
}

function onError( e: Event ) {
	console.log( 'on error: ', e )
}
</script>