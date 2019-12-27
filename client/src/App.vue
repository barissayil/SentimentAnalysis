<template>
	<div id="app">
		<Sentiment/>
	</div>
</template>

<script>
import Sentiment from './components/Sentiment.vue'

export default {
	name: 'app',
	components: {
		Sentiment
	}
}
</script>

<style>
#app {
	font-family: 'Avenir', Helvetica, Arial, sans-serif;
	-webkit-font-smoothing: antialiased;
	-moz-osx-font-smoothing: grayscale;
	text-align: center;
	color: #2c3e50;
	margin-top: 60px;
}
</style>
