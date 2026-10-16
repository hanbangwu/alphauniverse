<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import { getApp } from '$lib/state/app.svelte'
  import BookOpenIcon from '@lucide/svelte/icons/book-open'

  const app = getApp()
</script>

<footer
  class="flex shrink-0 flex-col items-start gap-2 p-4 font-mono text-xs text-muted-foreground"
>
  <span>{app.meta.galaxies} galaxies</span>
  <Button href="/docs" variant="outline" size="xs">
    <BookOpenIcon data-icon="inline-start" />
    Read the docs
  </Button>
</footer>
