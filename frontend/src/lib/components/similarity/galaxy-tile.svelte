<script lang="ts">
  import type { ImageSurvey } from '$lib/client'
  import DetailImage from '$lib/components/panels/detail-image.svelte'

  interface Props {
    galaxy: number
    survey: ImageSurvey
    onselect?: () => void
  }

  let { galaxy, survey, onselect }: Props = $props()
</script>

{#if onselect}
  <button
    type="button"
    onclick={onselect}
    aria-label="Search from galaxy {galaxy}"
    class="cursor-pointer rounded-lg ring-foreground outline-none hover:ring-2 focus-visible:ring-2"
  >
    <DetailImage {galaxy} {survey} />
  </button>
{:else}
  <DetailImage {galaxy} {survey} />
{/if}
