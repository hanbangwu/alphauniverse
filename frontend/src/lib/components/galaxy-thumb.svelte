<script lang="ts">
  import type { GetImageData } from '$lib/api'
  import { type ImageSurvey, client } from '$lib/client'
  import { Spinner } from '$lib/components/ui/spinner'
  import { SURVEYS } from '$lib/labels'

  interface Props {
    galaxy: number
    survey?: ImageSurvey
  }

  let { galaxy, survey = 'ls' }: Props = $props()

  let loaded = $state<number | null>(null)
  let failed = $state<number | null>(null)

  const src = $derived(
    client.buildUrl<GetImageData>({
      url: '/galaxy/{galaxy}/image/{survey}',
      path: { galaxy, survey }
    })
  )
</script>

{#if failed !== galaxy}
  <img
    {src}
    alt="{SURVEYS[survey].label} image of galaxy {galaxy}"
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
