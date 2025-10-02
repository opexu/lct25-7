<template>
    <div class="relative w-full h-full bg-black">
        <!-- Видео с камеры -->
        <video ref="videoRef" autoplay playsinline class="w-full h-full object-cover"
            :class="{ 'mirror': isFrontCamera }" />

        <!-- Оверлей с кнопкой -->
        <div class="absolute inset-0 flex flex-col justify-end pb-8">
            <!-- Кнопка фотографирования -->
            <div class="flex justify-center">
                <button @click="takePhoto"
                    class="w-20 h-20 bg-white rounded-full border-4 border-gray-300 flex items-center justify-center hover:bg-gray-100 transition-colors"
                    :disabled="isTakingPhoto">
                    <div class="w-16 h-16 bg-white rounded-full border-2 border-gray-400"></div>
                </button>
            </div>

            <!-- Переключение камеры -->
            <div class="flex justify-center mt-4">
                <button class="px-4 py-2 bg-gray-800 bg-opacity-50 text-white rounded-lg hover:bg-opacity-70 transition-colors"
                @click="switchCamera"
                >
                    📷 Переключить камеру
                </button>
            </div>
        </div>

        <div v-if="isTakingPhoto" class="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
            <div class="text-white text-lg">Создание фото...</div>
        </div>

        <WaitCameraView
        v-if="permissions === EPermissions.WAIT"
        />
        <NeedCameraView class="absolute left-0 top-0"
        v-else-if="permissions === EPermissions.ERR"
        @reqCameraPermissions="initCamera"
        />

    </div>
</template>

<script setup lang="ts">
import WaitCameraView from '@/views/WaitCameraView.vue'
import NeedCameraView from '@/views/NeedCameraView.vue'
import { useCamera } from '@/composables/useCamera'
import { EPermissions } from '@/scripts/types'
import { ref, onMounted, onUnmounted, useTemplateRef } from 'vue'
import { useVibration } from '@/composables/useVibration'

const emit = defineEmits( ['photoTaken'] )
const videoRef = useTemplateRef('videoRef')
const { permissions, isFrontCamera, initCamera, switchCamera, makePhoto, dispose } = useCamera( videoRef )
const isTakingPhoto = ref( false )


// Сделать фотографию
async function takePhoto() {
    if ( permissions.value !== EPermissions.OK || isTakingPhoto.value ) return
    isTakingPhoto.value = true
    try{
        const blob = await makePhoto()
        const base64 = await handlePhotoTaken( blob )
        emit('photoTaken', base64)
    }catch(e){
        console.error('Ошибка создания фото: ', e)
    }finally{
        isTakingPhoto.value = false
    }
    
}

onMounted( () => {
    initCamera()
} )

onUnmounted(() => {
    dispose()
})

async function handlePhotoTaken( blob: Blob ) {
    console.log( 'Фото сделано!', blob )
    const formData = new FormData()
    formData.append('file', blob)
    
    const url = getRembgUrl( true )
    const response = await fetch( url, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        throw new Error(`Ошибка: ${response.status}`);
    }
    console.log('response: ', response)
    const json = await response.json();
    console.log('json: ', json)
    return json.data
}

function validateOrigin(): string {
    const port = window.location.port
    let origin = window.location.origin
    return port ? origin.split(`:${port}`)[0] : origin
}

function getRembgUrl( isDev: boolean ): string {
    if( isDev ){
        return `http://${window.location.hostname}:8000/remove-background`
    }else{
        const origin = validateOrigin()
        return origin + `/api/remove-background`
    }
}
</script>

<style scoped>
.mirror {
    transform: scaleX(-1);
}

:deep(body) {
    margin: 0;
    padding: 0;
    overflow: hidden;
}
</style>