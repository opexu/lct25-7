<template>
    <div class="w-full h-full p-8 gap-4 flex flex-col items-center justify-center">
        <div class="w-full h-1/2">
            <img class="h-full object-cover" src="/child.gif" alt="gif" draggable="false">
        </div>
        <p class="text-primary font-bold text-4xl text-center">Ой! Кажется такой страницы не&nbsp;существует...</p>
        <router-link class="w-fit h-fit px-8 py-4 border rounded-2xl text-primary"
        to="/"
        >На главную</router-link>
    </div>
</template>

<script setup lang="ts">
</script>