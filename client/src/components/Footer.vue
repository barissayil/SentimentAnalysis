<template>
	<div class="footer">
  	<ul>
			Baris Sayil
			<li><a href="https://github.com/barissayil">Github</a></li>
			<li><a href="https://www.linkedin.com/in/barissayil/">LinkedIn</a></li>
			<li><a href="mailto:baris.sayil@insa-lyon.fr">Email</a></li>
  	</ul>
	</div>
</template>

<script>

</script>

<style scoped>
ul {
  display: inline-grid;
  grid-auto-flow: row;
  grid-gap: 24px;
  justify-items: center;
  margin: auto;
	color: white;
}

@media (min-width: 500px) {
  ul {
    grid-auto-flow: column;
  }
}

a {
  color: white;
  text-decoration: none;
  box-shadow: inset 0 -1px 0 hsla(0, 0%, 100%, 0.4);
}

a:hover {
  box-shadow: inset 0 -1.2em 0 hsla(0, 0%, 100%, 0.4);
}

/* li:last-child {
  grid-column: 1 / 2;
  grid-row: 1 / 2;
} */



div {
  display: flex;
  background-color: #002a38;
  line-height: 1.3;
  font-family: Menlo, monospace;
}


</style>
