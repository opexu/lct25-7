export function useVibration(){

    function vibrate( msArr: number[] ){
        if( 'vibrate' in navigator ){
            navigator.vibrate( msArr )
        }
    }

    return {
        vibrate
    }
}