<template>
    <div class="h-full bg-gray-100">
        <Camera v-if="!base64"
        @photoTaken="onPhoto"
        />
        <FinalImage v-else
        :src="base64"
        @back="base64 = ''"
        />
    </div>
</template>

<script setup lang="ts">
import Camera from '@/components/Camera.vue'
import FinalImage from '@/components/FinalImage.vue';
import { ref } from 'vue';
import CommonError from './CommonError.vue';

const base64 = ref('')
function onPhoto( src: string ){
    base64.value = src;
}
</script>