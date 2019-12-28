<template>
	<div class="sentiment">
		<h1>Sentiment Analysis!!!</h1>
		<textarea v-model="text" v-on:change="getSentiment"></textarea>
		<br>
		<button v-on:click="getSentiment">Submit</button>
		<br>
		<p style="white-space: pre-line;" v-if="reply">{{ sentiment }} with {{ probability }}% certainty.</p>
	</div>
</template>

<script>
import axios from "axios";

export default {
	name: 'Sentiment',
	props: {
		text: String,
		sentiment: String,
		probability: String,
		reply: Boolean
	},
	methods: {
		getSentiment() {
			axios.get('http://127.0.0.1:5000/', {
				params: {
					text: this.text
				}
			}).then((res)=>{
				if (this.text == ""){
					this.reply = false;
				} else {
					this.probability = res.data["probability"];
					this.sentiment = res.data["sentiment"];
					this.reply = true;
				}
			}).catch((err)=>{
				console.log(err);
			})
		}
	}
}
</script>

<style scoped>
h3 {
	margin: 40px 0 0;
}
ul {
	list-style-type: none;
	padding: 0;
}
li {
	display: inline-block;
	margin: 0 10px;
}
a {
	color: #42b983;
}
textarea {
	width: 400px;
	height: 100px;
}
</style>
