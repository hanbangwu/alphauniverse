<script lang="ts">
  import type { GetImageData, GetImageAionPredictionData } from '$lib/api'
  import { type ImageSurvey, client } from '$lib/client'
  import { Spinner } from '$lib/components/ui/spinner'
  import { coverageQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
    survey?: ImageSurvey
    predicted?: boolean
  }

  let { galaxy, survey = 'ls', predicted = false }: Props = $props()

  let loaded = $state<string | null>(null)
  let failed = $state<string | null>(null)

  const coverage = createQuery(() => coverageQuery(galaxy))

  const src = $derived(
    client.buildUrl<GetImageData | GetImageAionPredictionData>({
      url: predicted
        ? '/galaxy/{galaxy}/image/{survey}/aion_prediction'
        : '/galaxy/{galaxy}/image/{survey}',
      path: { galaxy, survey }
    })
  )
</script>

{#if failed !== src}
  <img
    {src}
    alt={`${predicted ? "AION's predicted " : ''}${SURVEYS[survey]} image of galaxy ${coverage.data?.object_id ?? ''}`}
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
