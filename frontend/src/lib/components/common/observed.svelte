<script lang="ts">
  import type { Galaxy } from '$lib/api'
  import { ANCHOR } from '$lib/data/modes'
  import { coverageQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { createQuery } from '@tanstack/svelte-query'
  import type { Snippet } from 'svelte'

  interface Props {
    galaxy: number
    survey: keyof Galaxy
    kind: 'image' | 'spectrum'
    children: Snippet
  }

  let { galaxy, survey, kind, children }: Props = $props()

  const coverage = createQuery(() => coverageQuery(galaxy))
</script>

{#if survey === ANCHOR || coverage.data?.[survey]}
  {@render children()}
{:else if coverage.data}
  <p class="text-xs text-muted-foreground">No {SURVEYS[survey].label} {kind}</p>
{/if}
