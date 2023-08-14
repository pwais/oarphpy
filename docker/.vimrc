" Indenting options
set paste        " Allow pasting from OS X
set autoindent   " Keep the indent level when hitting Return
set smartindent  " Use smart indenting (mostly useful for C/C++ files)
set cindent      " Don't indent Python really poorly
set tabstop=4    " Make tabs appear four spaces wide (default is 8 spaces)
set shiftwidth=4
"set noexpandtab  " Use hard tabs please! Watch out for files with soft tabs
                 " that don't have a modeline present, especially Python files.
set expandtab
set nocompatible " Don't care about VI-compatibility
set fo=tcoqan    " Options for formatting text (i.e. for use with gq)

" UI stuff
set showmatch   " Show matching parens as they come up
set ruler       " Show the column number in the status bar
set rulerformat=%30(%=\:b%n%y%m%r%w\ %l,%c%V\ %P%)
set incsearch   " Find as you type
set hlsearch
set wildmenu    " Get a cool menu for tab completing file names
set lz          " Don't redraw the screen in the middle of executing macros
set visualbell t_vb=
set smartcase   " Ignore case, unless caps are used in the search
behave xterm    " Just in case...
set lbr         " Wrap only at word boundaries (default is at any character)
set cursorline
set showcmd 	" show partial commands in status line and
				" selected characters/lines in visual mode

" Syntax Highlighting
syntax enable
colorscheme ron
let python_highlight_all = 1
let python_highlight_indent_errors = 0
let python_highlight_space_errors = 0
"autocmd FileType python source ~/.vim/python_fold.vim

"" Status line
set laststatus=2 "Always have a status line
set statusline=%2n:*%-32.32f%*\ \ %2*%r%m%*\ %=%y\ %15(%l/%L:%c\ (%2p%%)%)

" Python comment/uncomment
set backspace=indent,eol,start
