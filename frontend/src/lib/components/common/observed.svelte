<script lang="ts">
  import type { Galaxy } from '$lib/api'
  import { ANCHOR } from '$lib/data/modes'
  import { coverageQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { getView } from '$lib/state/app.svelte'
  import { createQuery } from '@tanstack/svelte-query'
  import type { Snippet } from 'svelte'

  interface Props {
    galaxy: number
    survey: keyof Galaxy
    kind: 'image' | 'spectrum'
    children: Snippet<[boolean]>
  }

  let { galaxy, survey, kind, children }: Props = $props()

  const view = getView()
  const coverage = createQuery(() => coverageQuery(galaxy))
  const matched = $derived(survey === ANCHOR || (coverage.data?.[survey] ?? false))
</script>

{#if matched || (coverage.data && view.predictions.value)}
  {@render children(matched)}
{:else if coverage.data}
  <p class="text-xs text-muted-foreground">No {SURVEYS[survey].label} {kind}</p>
{/if}
