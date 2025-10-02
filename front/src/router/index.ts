import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '@/views/HomeView.vue'
import NotFoundView from '@/views/NotFoundView.vue'
import CameraView from '@/views/CameraView.vue'
import CommonError from '@/views/CommonError.vue'

const router = createRouter( {
  history: createWebHistory( import.meta.env.BASE_URL ),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView,
    },
    {
      path: '/camera',
      name: 'camera',
      component: CameraView,
    },
    {
      path: '/error/:msg/:showMainBtn?',
      name: 'error',
      component: CommonError,
      props: true
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'NotFoundView',
      component: NotFoundView
    },
  ],
} )

export default router
