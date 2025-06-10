# Sample paragraph
paragraph = "You are Velgrith, exiled geneticist turned visionary conqueror.\n Armed with myth-tech and dark charisma, you return to reshape the world.\n Your first task: convert the defenders of Lysvaine into loyal, mythic beings...\n"

# Tokenize into words
words = paragraph.split()

# Tokenize each word in characters
characters = [list(word) for word in words]

# Output results
print("Words:", words)
print("Characters per word:", characters)

