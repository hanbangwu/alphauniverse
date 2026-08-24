<script lang="ts">
  import { Spinner } from '$lib/components/ui/spinner'
  import * as Table from '$lib/components/ui/table'
  import { coverageQuery } from '$lib/data/queries'
  import { SURVEYS } from '$lib/labels'
  import { getMeta } from '$lib/state/app.svelte'
  import CheckIcon from '@lucide/svelte/icons/check'
  import MinusIcon from '@lucide/svelte/icons/minus'
  import { createQuery } from '@tanstack/svelte-query'

  interface Props {
    galaxy: number
  }

  let { galaxy }: Props = $props()

  const meta = getMeta()

  const coverage = createQuery(() => coverageQuery(meta, galaxy))

  const surveys = $derived(coverage.data ?? [])
  const matched = $derived(surveys.filter((survey) => survey.matched).length)
</script>

<div class="flex flex-col gap-1">
  <div class="flex items-baseline justify-between">
    <span class="text-xs tracking-wider text-muted-foreground uppercase">Crossmatches</span>
    {#if coverage.isPending}
      <Spinner class="size-3 self-center text-muted-foreground" />
    {:else}
      <span class="text-xs text-muted-foreground tabular-nums">
        {matched} / {surveys.length}
      </span>
    {/if}
  </div>

  {#if surveys.length > 0}
    <Table.Root class="text-xs">
      <Table.Header>
        <Table.Row class="hover:bg-transparent">
          <Table.Head class="h-7 px-0 font-normal">Survey</Table.Head>
          <Table.Head class="h-7 px-0 text-right font-normal">Matched</Table.Head>
        </Table.Row>
      </Table.Header>
      <Table.Body>
        {#each surveys as { survey, matched } (survey)}
          <Table.Row class="last:border-0">
            <Table.Cell class="px-0 py-1.5">{SURVEYS[survey]?.label ?? survey}</Table.Cell>
            <Table.Cell class="px-0 py-1.5">
              <div class="flex justify-end">
                {#if matched}
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
