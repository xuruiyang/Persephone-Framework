grammar persephone;

/*------------------------------------------------------------------
* PARSER RULES
*------------------------------------------------------------------*/

main
   : game_definition game_play EOF
   ;

game_definition
	: GAME super_predicate_definition (super_predicate_definition)*
	;

super_predicate_definition
   : predicate_definition
   | gfp_definition
   | lfp_definition
   ;

predicate_definition
   : PREDICATE PRED_CONST LPAREN VAR (SEPARATOR VAR)* RPAREN EQUAL formula
   ;

gfp_definition
   : GFP PRED_CONST LPAREN VAR (SEPARATOR VAR)* RPAREN EQUAL formula
   ;

lfp_definition
   : LFP PRED_CONST LPAREN VAR (SEPARATOR VAR)* RPAREN EQUAL formula
   ;

game_play
   : PLAY PRED_CONST LPAREN term (SEPARATOR term)* RPAREN 
   ;

formula
	: quantifier LPAREN VAR SEPARATOR domain SEPARATOR formula RPAREN
   | bin_op LPAREN formula SEPARATOR formula RPAREN
	| NOT LPAREN formula RPAREN
	| PRED_CONST LPAREN term (SEPARATOR term)* RPAREN
   | ITE LPAREN cond SEPARATOR formula SEPARATOR formula RPAREN
	| TRUE
   | FALSE
	;

domain
   : DOM LPAREN term SEPARATOR term RPAREN
   | 's.getActions()'
   ;

cond
   : comp_op LPAREN term SEPARATOR term RPAREN
   | alg_bin_op LPAREN cond SEPARATOR cond RPAREN
   ;

alg_bin_op
   : AND
   | OR
   ;

comp_op
   : EQ
   | GE
   | LE
   | GT
   | LS
   ;

term
	: NUM
	| VAR (LPAREN VAR? RPAREN)?
	| algebra LPAREN term SEPARATOR term RPAREN
	;

algebra
   : Add
   | Sub
   ;

quantifier
   : EXISTS
   | FORALL
   ;

bin_op
	: AND
	| OR
	;

DOM
   :'Domain'
   ;

Add
   :'Add'
   ;
Sub
   :'Sub'
   ;
LPAREN
	:'('
	;
RPAREN
	:')'
	;
SEPARATOR
	:','
	;
EQUAL
	:'='
	;
ITE
   :'ITE'
   ;
NOT
	:'Not'
	;
FORALL
	:'Forall'
	;
EXISTS
	:'Exists'
	;
OR
	:'Or'
	;
AND
	:'And'
	;
EQ
   :'EQ'
   ;
GE
   :'GE'
   ;
LE
   :'LE'
   ;
GT
   :'GT'
   ;
LS
   :'LS'
   ;
TRUE
   :'TRUE'
   ;
FALSE
   :'FALSE'
   ;
GAME
   :'GAME'
   ;
PREDICATE
   :'PREDICATE'
   ;
GFP
   :'GFP'
   ;
LFP
   :'LFP'
   ;
PLAY
   :'PLAY'
   ;
ENDLINE
	:('\r'|'\n')+ -> skip
	;
WHITESPACE
	:(' '|'\t')+ -> skip
	;
PRED_CONST
   : [A-Z][A-Za-z_0-9]*
   ;
VAR
   : ([A-Za-z_0-9][A-Za-z_0-9]*)+('.'?[A-Za-z_0-9][A-Za-z_0-9]*)*
   ;
NUM
   : [0-9][0-9]*
   ;