<template>
	<div id="app">
		<Sentiment/>
		<Footer id="footer"/>
	</div>
</template>

<script>
import Sentiment from './components/Sentiment.vue'
import Footer from './components/Footer.vue'

export default {
	name: 'app',
	components: {
		Sentiment,
		Footer
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
#footer {
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  text-align: center;
}
</style>
