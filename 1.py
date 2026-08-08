Issues with custom keyboard

1.When we type on the custom keyboard some times it is clicking onto buttons behind the keyboard
2.when we start the session we start to type a name with the custom key board and
Shift to another app and comeback now we are seeing 2 keyboards both custom and default google keyboard

3.we want numbers also on to this alphabet keyboard

4.remove the thing saying typing username like this text

5.and when we press back button earlier when we are using keyboard the keyboard used to collapse(become invisible or gone)
now upon pressing back button it is going to previous page instead

6.for the numeric keypad keep this very simple numeric keypad and have 

7.match these dimensions for the keys on the keyboard

I compared the key geometry directly from both screenshots.
Dimensions to change
Your custom keys are currently approximately:
Letter key: 59 × 91 px
Gboard letter key: 60 × 71–72 px
So the biggest change is:
Reduce key height from ~91 px → ~72 px.
That's about 21% shorter.
Row positions
Your custom:
Row
Current Y
Current height
QWERTY
1049
91 px
ASDF
1150
91 px
ZXCV
1250
91 px
Bottom
1351
91 px
Gboard:
Row
Y
Height
QWERTY
1096
71 px
ASDF
1184
71–72 px
ZXCV
1273
71 px
Bottom
1362
~73 px
So the target should be approximately:
KEY HEIGHT       = 72 px
VERTICAL GAP     = 17 px
LETTER KEY WIDTH = 60 px
Horizontal dimensions
Gboard's normal letter keys are approximately:
width       ≈ 60 px
height      ≈ 71–72 px
horizontal gap ≈ 8 px
Your custom keyboard already has approximately the correct 60 px width, so don't significantly change the width.
Your current spacing is also close horizontally.
The main correction
Your current:
┌────────┐
│        │
│   Q    │   91 px
│        │
└────────┘
Change to:
┌────────┐
│        │
│   Q    │   72 px
└────────┘
And arrange the rows with approximately 17 px vertical separation.
Important
Don't change:
number of keys
key order
key functions
icons
toolbar
bottom-row functionality
keyboard logic
Only adjust:
key width + key height + row spacing + horizontal positioning + bottom-row dimensions

Yes — the curve/rounded corners are also important if you want the custom keyboard to visually match Gboard.
Looking at the two screenshots:
Gboard key corners: noticeably rounded, roughly 10–12 px radius
Your custom keys: also rounded, but they appear slightly more rounded/taller-looking because the keys themselves are taller.
Once you reduce the key height from ~91 px to ~72 px, the radius should also be reduced proportionally.
Recommended target
For the normal letter keys:
Width:         ~60 px
Height:        ~72 px
Corner radius: ~10 px
So don't keep something like:
60 × 72
radius = 18 px   ❌
because that will make the keys look excessively pill-shaped.
Use approximately:
60 × 72
radius = 10–11 px   ✅
Also important: keyboard outer edges
The keys at the extreme left/right don't need a different corner radius from the middle keys. The rounded appearance you're seeing comes primarily from each individual key's background shape.
So your dimensions should be tuned together:
key width → key height → corner radius → horizontal/vertical gap
rather than changing the radius independently.

For these what all files we need to change like should we change .kt files and .xnlml files both ?
