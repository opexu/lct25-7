import { defineStore } from "pinia";
import { readonly, ref } from "vue";

export const useDevice = defineStore('device-store', () => {
    const userAgent = navigator.userAgent;
    const isIOS = ref(/iPad|iPhone|iPod/.test(userAgent));
    const isAndroid = ref(/Android/.test(userAgent));
    const isMobile = ref(
        /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent) ||
        (navigator.maxTouchPoints && navigator.maxTouchPoints > 2)
    );
    const isTablet = ref(
        /iPad|Android(?!.*Mobile)|Tablet|Silk/i.test(userAgent) ||
        (navigator.maxTouchPoints && navigator.maxTouchPoints > 2 && /Mac|Windows/.test(navigator.platform))
    );
    const isDesktop = ref(!isMobile.value && !isTablet.value);
    return {
        isIOS: readonly( isIOS ),
        isDesktop: readonly( isDesktop ),
    }
})