import { EPermissions } from "@/scripts/types";
import { readonly, ref, type ShallowRef } from "vue";
import { useVibration } from "@/composables/useVibration";

type VRef = Readonly<ShallowRef<HTMLVideoElement | null>>
export function useCamera( videoRef: VRef ) {

    const { vibrate } = useVibration()

    const permissions = ref( EPermissions.WAIT )
    const stream = ref<MediaStream | null>( null )
    const isFrontCamera = ref( false )

    async function initCamera() {
        try {

            dispose()

            const constraints = {
                video: {
                    facingMode: isFrontCamera.value ? 'user' : 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: false
            }

            stream.value = await navigator.mediaDevices.getUserMedia( constraints )
            permissions.value = EPermissions.OK

            if ( videoRef.value ) {
                videoRef.value.srcObject = stream.value
            }
        } catch ( error ) {
            console.error( 'Ошибка доступа к камере:', error )
            alert( 'Не удалось получить доступ к камере. Проверьте разрешения.' )
            permissions.value = EPermissions.ERR
        } finally {
        }
    }

    async function switchCamera() {
        isFrontCamera.value = !isFrontCamera.value
        await initCamera()
    }

    async function makePhoto(): Promise<Blob> {
        return new Promise( ( res, rej ) => {
            try {
                vibrate( [ 50 ] )

                const canvas = document.createElement( 'canvas' )
                const context = canvas.getContext( '2d' )

                if ( !videoRef.value || !context ) return rej('!videoRef.value || !context');
                canvas.width = videoRef.value.videoWidth
                canvas.height = videoRef.value.videoHeight

                context.drawImage( videoRef.value, 0, 0, canvas.width, canvas.height )

                // const imageData = canvas.toDataURL( 'image/jpeg', 1.0 )

                // onSuccessCb && onSuccessCb( imageData )

                // downloadPhoto( imageData )
                canvas.toBlob( blob => {
                    return blob ? res( blob ) : rej('toBlob err')
                }, 'image/jpeg', 1.0 )
            } catch ( error ) {
                return rej(error)
            }
        } )

    }

    function downloadPhoto( imageData: string ) {
        const link = document.createElement( 'a' )
        link.download = `photo-${new Date().getTime()}.jpg`
        link.href = imageData
        link.click()
    }

    function dispose() {
        stream.value && stream.value.getTracks().forEach( track => track.stop() )
    }

    return {
        permissions: readonly( permissions ),
        isFrontCamera: readonly( isFrontCamera ),
        initCamera, switchCamera, makePhoto, dispose
    }
}