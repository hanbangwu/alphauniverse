<script lang="ts">
  import type { Meta } from '$lib/api'
  import { AppState, setApp } from '$lib/state/app.svelte'
  import type { MosaicState } from '$lib/state/mosaic.svelte'
  import { type Snippet, untrack } from 'svelte'

  interface Props {
    meta: Meta
    mosaic: MosaicState
    children: Snippet
  }

  let { meta, mosaic, children }: Props = $props()

  setApp(untrack(() => new AppState(meta, mosaic))).start()
</script>

{@render children()}
