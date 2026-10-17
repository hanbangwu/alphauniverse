<script lang="ts">
  import type { GetImageData, GetImageTemplateData } from '$lib/api'
  import { type ImageSurvey, client } from '$lib/client'
  import { Spinner } from '$lib/components/ui/spinner'
  import { SURVEYS } from '$lib/labels'

  interface Props {
    galaxy: number
    survey?: ImageSurvey
    predicted?: boolean
  }

  let { galaxy, survey = 'ls', predicted = false }: Props = $props()

  let loaded = $state<string | null>(null)
  let failed = $state<string | null>(null)

  const src = $derived(
    client.buildUrl<GetImageData | GetImageTemplateData>({
      url: predicted
        ? '/galaxy/{galaxy}/image/{survey}/template'
        : '/galaxy/{galaxy}/image/{survey}',
      path: { galaxy, survey }
    })
  )
</script>

{#if failed !== src}
  <img
    {src}
    alt={`${predicted ? "AION's predicted " : ''}${SURVEYS[survey].label} image of galaxy ${galaxy}`}
    loading="lazy"
    onload={() => (loaded = src)}
    onerror={() => (failed = src)}
    class="absolute inset-0 size-full object-fill"
    style:image-rendering="pixelated"
  />
{/if}

{#if predicted}
  <span
    class="pointer-events-none absolute top-1.5 left-1.5 rounded-sm bg-black/60 px-1 text-[10px] font-medium text-white"
  >
    AION
  </span>
{/if}

{#if failed !== src && loaded !== src}
  <div class="absolute inset-0 grid place-content-center">
    <Spinner class="size-6" />
  </div>
{/if}
