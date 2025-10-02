import { ref, computed, readonly } from 'vue'
import { defineStore } from 'pinia'

export const usePreloader = defineStore('preloader', () => {
  
  const active = ref(true)
  
  function setActive( value: boolean ){
    if( active.value === value ) return;
    active.value = value;
  }

  return { active: readonly( active ), setActive }
})
