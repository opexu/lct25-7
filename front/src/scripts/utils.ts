export function randInt( fromIncl: number, toIncl: number ): number {
  if (fromIncl > toIncl) throw new Error('fromIncl must be less than or equal to toIncl');
  return Math.floor(Math.random() * (toIncl - fromIncl + 1)) + fromIncl;
}

export function getRandomFromRange( range: number[] ): number {
    const index = randInt( 0, range.length - 1 )
    return range[ index ]
}

export function genRange( from: number, to: number ): number[] {
    const length = to - from + 1;
    return Array.from({length}, (_, i) => i + from)
}