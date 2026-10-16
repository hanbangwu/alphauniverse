<script lang="ts">
  import type { GetImageData, GetImageTemplateData } from '$lib/api'
  import { client } from '$lib/client'
  import { Spinner } from '$lib/components/ui/spinner'

  interface Props {
    galaxy: number
    predicted?: boolean
  }

  let { galaxy, predicted = false }: Props = $props()

  let loaded = $state<string | null>(null)
  let failed = $state<string | null>(null)

  const src = $derived(
    predicted
      ? client.buildUrl<GetImageTemplateData>({
          url: '/galaxy/{galaxy}/image/template',
          path: { galaxy }
        })
      : client.buildUrl<GetImageData>({ url: '/galaxy/{galaxy}/image', path: { galaxy } })
  )
</script>

{#if failed !== src}
  <img
    {src}
    alt={`${predicted ? "AION's predicted " : ''}Legacy Survey image of galaxy ${galaxy}`}
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
