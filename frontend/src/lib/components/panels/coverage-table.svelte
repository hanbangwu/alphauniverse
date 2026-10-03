<script lang="ts">
  import type { Galaxy } from '$lib/api'
  import { Spinner } from '$lib/components/ui/spinner'
  import * as Table from '$lib/components/ui/table'
  import { coverageQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import CheckIcon from '@lucide/svelte/icons/check'
  import MinusIcon from '@lucide/svelte/icons/minus'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const coverage = createQuery(() => coverageQuery(galaxy))
</script>

<div class="flex flex-col gap-1">
  <div class="flex items-baseline justify-between">
    <span class="text-xs tracking-wider text-muted-foreground uppercase">Crossmatches</span>
    {#if coverage.isPending}
      <Spinner class="size-3 self-center text-muted-foreground" />
    {/if}
  </div>

  {#if coverage.data}
    <Table.Root class="text-xs">
      <Table.Header>
        <Table.Row class="hover:bg-transparent">
          <Table.Head class="h-7 px-0 font-normal">Survey</Table.Head>
          <Table.Head class="h-7 px-0 text-right font-normal">Matched</Table.Head>
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {#each Object.entries(SURVEYS) as [survey, { label }] (survey)}
          <Table.Row class="last:border-0">
            <Table.Cell class="px-0 py-1.5">{label}</Table.Cell>
            <Table.Cell class="px-0 py-1.5">
              <div class="flex justify-end">
                {#if coverage.data[survey as keyof Galaxy]}
                  <CheckIcon class="size-3.5" aria-label="matched" />
                {:else}
                  <MinusIcon class="size-3.5 text-muted-foreground/50" aria-label="not matched" />
                {/if}
              </div>
            </Table.Cell>
          </Table.Row>
        {/each}
      </Table.Body>
    </Table.Root>
  {/if}
</div>
