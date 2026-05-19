{ % macro generate_schema_name(custom_schema_name, node) - % } { % - if custom_schema_name is none - % } { # No +schema defined on this model — fall back to the profiles.yml schema #}
{ { target.schema } } { % -
else - % } { # +schema IS defined — use it as the complete schema name, no concatenation #}
{ { custom_schema_name | trim } } { % - endif - % } { % - endmacro % }