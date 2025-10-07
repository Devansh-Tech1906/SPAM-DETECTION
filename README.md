Spam & Ham Text Detector 💌
Ever wonder how your phone knows a "Congratulations! You've won a new iPhone!" text is probably a scam? 
This project is a tiny, personal peek behind that curtain! It's a simple, but effective, machine learning model that can tell the difference between a real text ("ham") and a spam message trying to sell you something.

It's built with Python and a whole lot of heart

💡 The Big Idea
The goal here was to create a reliable text classifier from scratch. Instead of just building a model, I wanted to understand why it works. So, the process was a lot like being a detective:

1> Cleaning Up the Clutter: First, we take all the raw, messy text and tidy it up. We get rid of punctuation, make everything lowercase, and toss out common, unhelpful words like "the" or "and."

2> Turning Words into Numbers: Computers don't speak English, so we have to translate. We use a cool technique called TF-IDF to turn each message into a series of numbers that represent how important each word is. It's like giving each word a score!

3> Teaching the Brain: Then, we feed these numbers into a Multinomial Naive Bayes model. It's a powerful and efficient model that learns the patterns and tells us, "Hey, these words look a lot like a spam message."

4> Making a Guess: Once the model is trained, we can give it any new text message, and it will take a pretty good guess at whether it's spam or not.

🚀 Let's Get It Running
Ready to see it in action?
https://spam-detection-7y3v.onrender.com
