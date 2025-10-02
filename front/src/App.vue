<template>
	<div class="w-full h-full flex flex-col overflow-hidden">
		<router-view v-slot=" { Component, route } ">
			<Transition name="fade" mode="out-in" appear>
				<component :is="Component" :key="route.path" />
			</Transition>
		</router-view>
	</div>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'

</script>

<style scoped></style>
