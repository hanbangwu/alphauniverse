<script lang="ts">
  import type { GetImageData } from '$lib/api'
  import { client } from '$lib/client'
  import { Spinner } from '$lib/components/ui/spinner'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  let loaded = $state<number | null>(null)
  let failed = $state<number | null>(null)

  const src = $derived(
    client.buildUrl<GetImageData>({ url: '/galaxies/{galaxy}/image.png', path: { galaxy } })
  )
</script>

{#if failed !== galaxy}
  <img
    {src}
    alt="Legacy Survey cutout of galaxy {galaxy}"
    loading="lazy"
    onload={() => (loaded = galaxy)}
    onerror={() => (failed = galaxy)}
    class="absolute inset-0 size-full object-fill"
    style:image-rendering="pixelated"
  />
{/if}

{#if failed !== galaxy && loaded !== galaxy}
  <div class="absolute inset-0 grid place-content-center">
    <Spinner class="size-6" />
  </div>
{/if}
