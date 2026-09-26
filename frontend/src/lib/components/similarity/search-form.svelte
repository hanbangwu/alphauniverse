<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import MatchCount from './match-count.svelte'
  import { getSimilarity } from './similarity.svelte'

  const similarity = getSimilarity()
</script>

<form
  class="flex flex-wrap items-center gap-x-4 gap-y-2"
  onsubmit={(event) => {
    event.preventDefault()
    similarity.submit()
  }}
>
  <MatchCount />
  <Button type="submit" size="sm" disabled={similarity.draft === null}>Search</Button>
</form>
