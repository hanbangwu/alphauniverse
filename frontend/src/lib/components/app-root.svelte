<script lang="ts">
  import type { Meta } from '$lib/api'
  import { AppState, setApp } from '$lib/state/app.svelte'
  import { type Snippet, untrack } from 'svelte'

  interface Props {
    meta: Meta
    children: Snippet
  }

  let { meta, children }: Props = $props()

  setApp(untrack(() => new AppState(meta))).start()
</script>

{@render children()}
