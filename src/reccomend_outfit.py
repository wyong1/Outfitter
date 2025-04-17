import random

closet = {
    'tops': ['white shirt', 'black t-shirt', 'blue blouse'],
    'bottoms': ['jeans', 'black skirt', 'khaki pants'],
    'shoes': ['sneakers', 'heels', 'loafers']
}

def recommend_outfit():
    return {
        'top': random.choice(closet['tops']),
        'bottom': random.choice(closet['bottoms']),
        'shoes': random.choice(closet['shoes'])
    }

if __name__ == "__main__":
    outfit = recommend_outfit()
    print(f"Today's outfit: {outfit['top']}, {outfit['bottom']}, and {outfit['shoes']}")
