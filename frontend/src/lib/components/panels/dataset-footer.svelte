<script lang="ts">
  import { Button } from '$lib/components/ui/button'
  import BookOpenIcon from '@lucide/svelte/icons/book-open'
</script>

<footer
  class="flex shrink-0 flex-col items-start gap-2 p-4 font-mono text-xs text-muted-foreground"
>
  <Button href="/docs" variant="outline" size="xs">
    <BookOpenIcon data-icon="inline-start" />
    Read the docs
  </Button>
</footer>
